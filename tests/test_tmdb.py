import asyncio
import time

import httpx
import pytest

from lbxd.tmdb import Metadata, MetadataCache, TMDBClient, _parse

PARASITE = {
    "title": "Parasite",
    "overview": "Greed and class discrimination threaten a newly formed symbiosis.",
    "poster_path": "/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
    "backdrop_path": "/TU9NIjwzjoKPwQHoHshkFcQUCG.jpg",
    "runtime": 133,
    "vote_average": 8.5,
    "release_date": "2019-05-30",
    "genres": [{"name": "Comedy"}, {"name": "Thriller"}],
    "credits": {"crew": [
        {"job": "Director", "name": "Bong Joon-ho"},
        {"job": "Editor", "name": "Yang Jin-mo"},
    ]},
}


def transport(handler):
    return httpx.MockTransport(handler)


def test_parse_pulls_the_fields_a_page_needs():
    meta = _parse(496243, PARASITE)

    assert meta.title == "Parasite"
    assert meta.director == "Bong Joon-ho"
    assert meta.runtime == 133
    assert meta.genres == ["Comedy", "Thriller"]
    assert meta.poster_url == (
        "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg"
    )


def test_parse_keeps_both_names_for_a_co_directed_film():
    payload = dict(PARASITE, credits={"crew": [
        {"job": "Director", "name": "Joel Coen"},
        {"job": "Director", "name": "Ethan Coen"},
    ]})
    assert _parse(1, payload).director == "Joel Coen & Ethan Coen"


def test_missing_images_become_null_not_a_broken_url():
    payload = dict(PARASITE, poster_path=None, backdrop_path=None)
    meta = _parse(1, payload)
    assert meta.poster_url is None and meta.backdrop_url is None


def test_disabled_client_fetches_nothing(tmp_path):
    client = TMDBClient(cache=MetadataCache(tmp_path / "c.sqlite"))
    assert not client.enabled
    assert asyncio.run(client.fetch_many([496243])) == {}


def test_fetches_and_then_serves_from_cache(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, json=PARASITE)

    cache = MetadataCache(tmp_path / "c.sqlite")
    client = TMDBClient(api_key="k", cache=cache, transport=transport(handler))

    first = asyncio.run(client.fetch_many([496243]))
    assert first[496243].title == "Parasite"
    assert len(calls) == 1

    # A second client sharing the cache must not touch the network at all.
    again = TMDBClient(api_key="k", cache=cache, transport=transport(handler))
    second = asyncio.run(again.fetch_many([496243]))
    assert second[496243].title == "Parasite"
    assert len(calls) == 1


def test_v3_key_goes_in_the_query_and_v4_token_in_the_header(tmp_path):
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=PARASITE)

    cache = MetadataCache(tmp_path / "a.sqlite")
    asyncio.run(TMDBClient(api_key="abc", cache=cache,
                           transport=transport(handler)).fetch_many([1]))
    assert seen["params"]["api_key"] == "abc" and seen["auth"] is None

    cache2 = MetadataCache(tmp_path / "b.sqlite")
    asyncio.run(TMDBClient(read_token="tok", cache=cache2,
                           transport=transport(handler)).fetch_many([2]))
    assert seen["auth"] == "Bearer tok" and "api_key" not in seen["params"]


def test_a_404_is_remembered_so_we_stop_asking(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, json={"status_message": "Not found"})

    cache = MetadataCache(tmp_path / "c.sqlite")
    client = TMDBClient(api_key="k", cache=cache, transport=transport(handler))

    assert asyncio.run(client.fetch_many([999999])) == {999999: None}
    assert asyncio.run(client.fetch_many([999999])) == {999999: None}
    assert len(calls) == 1          # the miss was cached, not re-requested


def test_rate_limiting_is_retried_and_honours_retry_after(tmp_path):
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json=PARASITE),
    ]

    def handler(request):
        return responses.pop(0)

    client = TMDBClient(api_key="k", cache=MetadataCache(tmp_path / "c.sqlite"),
                        transport=transport(handler))
    result = asyncio.run(client.fetch_many([496243]))
    assert result[496243].title == "Parasite"
    assert responses == []


def test_a_failure_degrades_to_none_instead_of_raising(tmp_path):
    def handler(request):
        raise httpx.ConnectError("network down")

    client = TMDBClient(api_key="k", cache=MetadataCache(tmp_path / "c.sqlite"),
                        transport=transport(handler))
    # The recommendation list is the product; a poster is decoration.
    assert asyncio.run(client.fetch_many([496243])) == {496243: None}


def test_bad_credentials_are_not_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"status_message": "Invalid API key"})

    client = TMDBClient(api_key="wrong", cache=MetadataCache(tmp_path / "c.sqlite"),
                        transport=transport(handler))
    asyncio.run(client.fetch_many([1]))
    assert len(calls) == 1          # retrying a bad key just wastes time


def test_duplicate_ids_are_requested_once(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=PARASITE)

    client = TMDBClient(api_key="k", cache=MetadataCache(tmp_path / "c.sqlite"),
                        transport=transport(handler))
    result = asyncio.run(client.fetch_many([496243, 496243, 496243]))
    assert len(calls) == 1 and len(result) == 1


def test_stale_cache_entries_are_refetched(tmp_path):
    cache = MetadataCache(tmp_path / "c.sqlite")
    cache.put_many({7: Metadata(tmdb_id=7, title="Old")})
    cache._db.execute("UPDATE metadata SET fetched = ?", (time.time() - 10**9,))
    cache._db.commit()

    def handler(request):
        return httpx.Response(200, json=dict(PARASITE, title="Fresh"))

    client = TMDBClient(api_key="k", cache=cache, transport=transport(handler))
    assert asyncio.run(client.fetch_many([7]))[7].title == "Fresh"


def test_concurrency_is_capped(tmp_path):
    """A polite client: many ids in flight, but not all at once."""
    live = 0
    peak = 0

    async def handler(request):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return httpx.Response(200, json=PARASITE)

    client = TMDBClient(api_key="k", cache=MetadataCache(tmp_path / "c.sqlite"),
                        concurrency=3, transport=httpx.MockTransport(handler))
    asyncio.run(client.fetch_many(list(range(1, 16))))
    assert peak <= 3


def test_empty_input_short_circuits(tmp_path):
    client = TMDBClient(api_key="k", cache=MetadataCache(tmp_path / "c.sqlite"))
    assert asyncio.run(client.fetch_many([])) == {}
    assert asyncio.run(client.fetch_many([None, 0])) == {}
